from nltk.corpus import verbnet as vn

class VerbNetManager:
    def getClasses(self, verb):
        return vn.classids(verb)

    def getThematicRoles(self, verb):
        thematicRoles = []
        for verbClass in self.getClasses(verb):
            for themrole in vn.vnclass(verbClass).findall('THEMROLES/THEMROLE'):
                role = themrole.attrib['type']
                for selrestr in themrole.findall('SELRESTRS/SELRESTR'):
                    role += '[%(Value)s%(type)s]' % selrestr.attrib
                thematicRoles.append(role)

        return thematicRoles
